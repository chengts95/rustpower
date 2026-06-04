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

                        // Check if it's Option<T>
                        if let syn::Type::Path(type_path) = ty {
                            if type_path.path.segments.len() == 1
                                && type_path.path.segments[0].ident == "Option"
                            {
                                // It is Option<T>, only insert if Some
                                quote! {
                                    if let Some(val) = self.#fname {
                                        buffer.insert(world, entity, val);
                                    }
                                }
                            } else {
                                // Standard field, always insert
                                quote! {
                                    buffer.insert(world, entity, self.#fname);
                                }
                            }
                        } else {
                            quote! {
                                buffer.insert(world, entity, self.#fname);
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
        impl crate::bevy_cmdbuffer::buffer::DeferBundle for #name {
            fn insert_into(self, buffer: &mut crate::bevy_cmdbuffer::buffer::HarvardCommandBuffer, world: &mut bevy_ecs::prelude::World, entity: bevy_ecs::prelude::Entity) {
                #(#field_insertions)*
            }
        }
    };

    TokenStream::from(expanded)
}
