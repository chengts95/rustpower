use proc_macro::TokenStream;
use quote::quote;
use syn::*;
#[proc_macro_derive(DeferBundle)]
pub fn derive_defer_bundle(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let field_insertions = match input.data {
        syn::Data::Struct(data) => match data.fields {
            syn::Fields::Named(ref fields) => {
                fields
                    .named
                    .iter()
                    .map(|f| {
                        let fname = f.ident.as_ref().unwrap();
                        let ty = &f.ty;

                        // 判断是不是 Option<_>
                        if let syn::Type::Path(type_path) = ty {
                            if type_path.path.segments.len() == 1
                                && type_path.path.segments[0].ident == "Option"
                            {
                                // 是 Option<T>
                                quote! {
                                    if let Some(val) = &self.#fname {
                                        builder.insert(val.clone());
                                    }
                                }
                            } else {
                                // 不是 Option<T>
                                quote! {
                                    builder.insert(self.#fname.clone());
                                }
                            }
                        } else {
                            // fallback: treat as always insert
                            quote! {
                                builder.insert(self.#fname.clone());
                            }
                        }
                    })
                    .collect::<Vec<_>>()
            }
            _ => panic!("DeferBundle can only be derived for named struct"),
        },
        _ => panic!("DeferBundle only supports structs"),
    };

    let expanded = quote! {
        impl DeferBundle for #name {
            fn insert_to(self, builder: &mut DeferredBundleBuilder) {
                #(#field_insertions)*
            }
        }
    };

    TokenStream::from(expanded)
}
